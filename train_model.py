#!/usr/bin/env python3
"""
Training script for the Isolation Forest anomaly detector.

Usage:
    python train_model.py --duration 60 --contamination 0.01
    python train_model.py --from-file data/training/baseline.csv
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model_trainer import train_and_save_model, ModelTrainer
from src.logger_setup import logger


def main():
    parser = argparse.ArgumentParser(
        description="Train the Isolation Forest anomaly detection model"
    )
    
    # Training data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--duration',
        type=int,
        help='Collect baseline traffic for N seconds'
    )
    data_group.add_argument(
        '--from-file',
        type=str,
        help='Load training data from CSV file'
    )
    
    # Model parameters
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.01,
        help='Expected proportion of anomalies (default: 0.01)'
    )
    parser.add_argument(
        '--version',
        type=int,
        default=1,
        help='Model version number (default: 1)'
    )
    parser.add_argument(
        '--no-save-data',
        action='store_true',
        help='Do not save collected training data to CSV'
    )
    parser.add_argument(
        '--interface',
        type=str,
        default=None,
        help='Network interface to monitor (default: all interfaces)'
    )
    
    args = parser.parse_args()
    
    try:
        trainer = ModelTrainer()
        
        if args.duration:
            # Collect live traffic
            logger.info(f"Collecting baseline traffic for {args.duration} seconds...")
            n_samples = trainer.collect_baseline_traffic(
                duration=args.duration,
                interface=args.interface
            )
            logger.info(f"Collected {n_samples} samples")
            
            if n_samples < 10:
                logger.error("Insufficient training samples collected. Need at least 10.")
                return 1
            
            # Save training data
            if not args.no_save_data:
                trainer.save_training_data()
        
        else:
            # Load from file
            logger.info(f"Loading training data from {args.from_file}")
            trainer.load_training_data(args.from_file)
        
        # Train model
        logger.info("Training Isolation Forest model...")
        stats = trainer.train_model(contamination=args.contamination)
        
        # Print statistics
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Samples:          {stats['n_samples']}")
        print(f"Features:         {stats['n_features']}")
        print(f"Training time:    {stats['training_time_seconds']:.2f}s")
        print(f"Anomalies found:  {stats['n_anomalies_detected']} ({stats['anomaly_rate']:.2%})")
        print(f"Score range:      [{stats['score_min']:.3f}, {stats['score_max']:.3f}]")
        print(f"Score mean/std:   {stats['score_mean']:.3f} ± {stats['score_std']:.3f}")
        print("="*60)
        
        # Save model
        logger.info(f"Saving model version {args.version}...")
        trainer.save_model(version=args.version)
        
        print(f"\n✅ Model saved successfully (version {args.version})")
        print(f"   Model path: models/isolation_forest_v{args.version}.joblib")
        print(f"   Scaler path: models/scaler_v{args.version}.joblib")
        print("\nYou can now run the anomaly detector with ML detection enabled.")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
