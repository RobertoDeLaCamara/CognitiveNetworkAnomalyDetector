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
from pathlib import Path

# Add src to path securely
script_dir = Path(__file__).parent.resolve()
src_path = script_dir / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    print("Error: src directory not found")
    sys.exit(1)

try:
    from src.model_trainer import ModelTrainer
    from src.logger_setup import logger
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def validate_args(args):
    """Validate command line arguments.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    if args.duration is not None:
        if args.duration <= 0 or args.duration > 3600:  # Max 1 hour
            raise ValueError("Duration must be between 1 and 3600 seconds")
    
    if args.from_file is not None:
        file_path = Path(args.from_file).resolve()
        
        # Check for path traversal attempts
        if ".." in args.from_file or "~" in args.from_file:
            raise ValueError("Path traversal detected in file path")
        
        if not file_path.exists():
            raise ValueError(f"Training file does not exist: {args.from_file}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {args.from_file}")
        if file_path.suffix.lower() != '.csv':
            raise ValueError("Training file must be a CSV file")
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024
        if file_path.stat().st_size > max_size:
            raise ValueError(f"Training file too large (max 100MB)")
    
    if not (0.001 <= args.contamination <= 0.5):
        raise ValueError("Contamination must be between 0.001 and 0.5")
    
    if args.version <= 0 or args.version > 999:
        raise ValueError("Version must be between 1 and 999")
    
    # Validate string inputs for injection attacks
    if args.experiment_name:
        if len(args.experiment_name) > 100 or not args.experiment_name.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Invalid experiment name (use alphanumeric, dash, underscore only)")
    
    if args.run_name:
        if len(args.run_name) > 100 or not args.run_name.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Invalid run name (use alphanumeric, dash, underscore only)")
    
    if args.interface:
        # Basic validation for network interface names
        if len(args.interface) > 20 or not args.interface.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Invalid interface name")

def main():
    parser = argparse.ArgumentParser(
        description="Train the Isolation Forest anomaly detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --duration 60 --contamination 0.01
  %(prog)s --from-file data/training/baseline.csv --version 2
        """
    )
    
    # Training data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--duration',
        type=int,
        metavar='SECONDS',
        help='Collect baseline traffic for N seconds (1-3600)'
    )
    data_group.add_argument(
        '--from-file',
        type=str,
        metavar='PATH',
        help='Load training data from CSV file'
    )
    
    # Model parameters
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.01,
        metavar='RATIO',
        help='Expected proportion of anomalies (0.001-0.5, default: 0.01)'
    )
    parser.add_argument(
        '--version',
        type=int,
        default=1,
        metavar='N',
        help='Model version number (1-999, default: 1)'
    )
    parser.add_argument(
        '--no-save-data',
        action='store_true',
        help='Do not save collected training data to CSV'
    )
    parser.add_argument(
        '--interface',
        type=str,
        metavar='IFACE',
        help='Network interface to monitor (default: all interfaces)'
    )
    
    # MLflow options
    parser.add_argument(
        '--experiment-name',
        type=str,
        metavar='NAME',
        help='MLflow experiment name (default: cognitive-anomaly-detector)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        metavar='NAME',
        help='MLflow run name (default: auto-generated)'
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking for this run'
    )
    
    try:
        args = parser.parse_args()
        validate_args(args)
    except ValueError as e:
        parser.error(str(e))
    
    try:
        # Check if running with appropriate privileges for packet capture
        if args.duration and os.name == 'posix' and os.geteuid() != 0:
            logger.warning("Not running as root. Packet capture may be limited.")
            logger.warning("Consider running with sudo for full network access.")
        
        # Initialize trainer with MLflow configuration
        enable_mlflow = not args.no_mlflow
        trainer = ModelTrainer(enable_mlflow=enable_mlflow)
        
        if trainer.mlflow_enabled:
            logger.info("MLflow tracking enabled")
            if args.experiment_name:
                logger.info(f"Experiment: {args.experiment_name}")
            if args.run_name:
                logger.info(f"Run name: {args.run_name}")
        
        if args.duration:
            # Collect live traffic
            logger.info(f"Collecting baseline traffic for {args.duration} seconds...")
            logger.info(f"Collecting baseline traffic for {args.duration} seconds...")
            logger.info("Press Ctrl+C to stop early if needed.")
            
            try:
                n_samples = trainer.collect_baseline_traffic(
                    duration=args.duration,
                    interface=args.interface
                )
            except KeyboardInterrupt:
                logger.info("Collection interrupted by user")
                logger.info("Training data collection interrupted by user")
                return 1
            except PermissionError:
                logger.error("Permission denied for packet capture")
                logger.error("Try running with sudo or check network interface permissions")
                return 1
            
            logger.info(f"Collected {n_samples} samples")
            logger.info(f"Collected {n_samples} samples")
            
            if n_samples < 10:
                logger.error("Insufficient training samples collected. Need at least 10.")
                logger.error("Insufficient training samples collected. Need at least 10.")
                logger.error("Try increasing the duration or generating more network traffic.")
                return 1
            
            # Save training data
            if not args.no_save_data:
                try:
                    trainer.save_training_data()
                    logger.info("Training data saved successfully")
                except Exception as save_e:
                    logger.warning(f"Failed to save training data: {save_e}")
                    logger.warning(f"Failed to save training data: {save_e}")
        
        else:
            # Load from file
            logger.info(f"Loading training data from {args.from_file}")
            logger.info(f"Loading training data from {args.from_file}")
            try:
                trainer.load_training_data(args.from_file)
            except Exception as load_e:
                logger.error(f"Failed to load training data: {load_e}")
                logger.error(f"Failed to load training data: {load_e}")
                return 1
        
        # Train model
        logger.info("Training Isolation Forest model...")
        logger.info("Training Isolation Forest model...")
        
        try:
            stats = trainer.train_model(
                contamination=args.contamination,
                experiment_name=args.experiment_name,
                run_name=args.run_name
            )
        except Exception as train_e:
            logger.error(f"Model training failed: {train_e}")
            logger.error(f"Model training failed: {train_e}")
            return 1
        
        # Print statistics
        logger.info("%s", "\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("%s", "="*60)
        logger.info(f"Samples:          {stats['n_samples']}")
        logger.info(f"Features:         {stats['n_features']}")
        logger.info(f"Training time:    {stats['training_time_seconds']:.2f}s")
        logger.info(f"Anomalies found:  {stats['n_anomalies_detected']} ({stats['anomaly_rate']:.2%})")
        logger.info(f"Score range:      [{stats['score_min']:.3f}, {stats['score_max']:.3f}]")
        logger.info(f"Score mean/std:   {stats['score_mean']:.3f} ± {stats['score_std']:.3f}")
        logger.info("%s", "="*60)
        
        # Save model
        logger.info(f"Saving model version {args.version}...")
        logger.info(f"Saving model version {args.version}...")
        
        try:
            # Extract run_id if available
            run_id = stats.get('mlflow_run_id')
            trainer.save_model(version=args.version, run_id=run_id)
        except Exception as save_e:
            logger.error(f"Failed to save model: {save_e}")
            logger.error(f"Failed to save model: {save_e}")
            return 1
        
        logger.info(f"Model saved successfully (version {args.version})")
        logger.info(f"   Model path: models/isolation_forest_v{args.version}.joblib")
        logger.info(f"   Scaler path: models/scaler_v{args.version}.joblib")
        
        # Print MLflow info if enabled
        if trainer.mlflow_enabled and 'mlflow_run_id' in stats:
            logger.info(f"   MLflow run ID: {stats['mlflow_run_id']}")
            logger.info(f"   View in MLflow UI: mlflow ui --backend-store-uri file://{os.getcwd()}/.mlflow/mlruns")
        
        logger.info("You can now run the anomaly detector with ML detection enabled.")
        
        return 0
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
