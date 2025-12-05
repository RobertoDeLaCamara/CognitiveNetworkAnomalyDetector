#!/usr/bin/env python3
"""
Demo script to generate synthetic network traffic for training the anomaly detector.
This creates realistic-looking traffic patterns without actually sending packets.
"""

import numpy as np
from src.feature_extractor import FeatureExtractor
from src.model_trainer import ModelTrainer
from src.ml_config import N_FEATURES

def generate_synthetic_training_data(n_ips=20, packets_per_ip=10):
    """Generate synthetic training data that mimics normal network traffic.
    
    Args:
        n_ips: Number of unique IP addresses to simulate
        packets_per_ip: Number of packets per IP
    
    Returns:
        Numpy array of training features
    """
    print(f"Generating synthetic training data for {n_ips} IPs...")
    
    np.random.seed(42)
    
    # Simulate normal network traffic patterns
    training_features = []
    
    for ip_idx in range(n_ips):
        # Statistical features (6)
        packets_per_second = np.random.uniform(0.5, 5.0)  # Low to moderate traffic
        bytes_per_second = packets_per_second * np.random.uniform(500, 1500)
        avg_packet_size = np.random.uniform(500, 1500)
        packet_size_variance = np.random.uniform(100, 500)
        total_packets = packets_per_ip
        total_bytes = total_packets * avg_packet_size
        
        # Temporal features (4)
        inter_arrival_mean = 1.0 / packets_per_second
        inter_arrival_std = inter_arrival_mean * 0.2
        burst_rate = np.random.uniform(0.5, 2.0)
        session_duration = np.random.uniform(10, 60)
        
        # Protocol features (3) - mostly TCP and UDP
        tcp_ratio = np.random.uniform(0.6, 0.9)
        udp_ratio = np.random.uniform(0.05, 0.3)
        icmp_ratio = max(0, 1 - tcp_ratio - udp_ratio)
        
        # Normalize to sum to 1
        total = tcp_ratio + udp_ratio + icmp_ratio
        tcp_ratio /= total
        udp_ratio /= total
        icmp_ratio /= total
        
        # Port features (2)
        unique_ports = np.random.randint(1, 5)
        uncommon_port_ratio = np.random.uniform(0.0, 0.3)
        
        # Payload features (3)
        payload_entropy = np.random.uniform(3.0, 6.0)  # Normal text/data entropy
        avg_payload_size = np.random.uniform(100, 800)
        payload_size_variance = np.random.uniform(50, 300)
        
        # Construct feature vector
        features = [
            packets_per_second,
            bytes_per_second,
            avg_packet_size,
            packet_size_variance,
            total_packets,
            total_bytes,
            inter_arrival_mean,
            inter_arrival_std,
            burst_rate,
            session_duration,
            tcp_ratio,
            udp_ratio,
            icmp_ratio,
            unique_ports,
            uncommon_port_ratio,
            payload_entropy,
            avg_payload_size,
            payload_size_variance
        ]
        
        training_features.append(features)
    
    return np.array(training_features)


def main():
    """Generate synthetic data and train the model."""
    print("=" * 60)
    print("DEMO: Training Anomaly Detector with Synthetic Data")
    print("=" * 60)
    print()
    
    # Generate synthetic training data
    training_data = generate_synthetic_training_data(n_ips=100, packets_per_ip=20)
    
    print(f"Generated {len(training_data)} training samples")
    print(f"Feature shape: {training_data.shape}")
    print()
    
    # Initialize trainer and detector
    trainer = ModelTrainer()
    
    # Directly set the training features (bypass packet collection)
    trainer.training_features = training_data
    
    # Train model
    print("Training Isolation Forest model...")
    stats = trainer.train_model(contamination=0.01)
    
    # Print training statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Samples:          {stats['n_samples']}")
    print(f"Features:         {stats['n_features']}")
    print(f"Training time:    {stats['training_time_seconds']:.2f}s")
    print(f"Anomalies found:  {stats['n_anomalies_detected']} ({stats['anomaly_rate']:.2%})")
    print(f"Score range:      [{stats['score_min']:.3f}, {stats['score_max']:.3f}]")
    print(f"Score mean/std:   {stats['score_mean']:.3f} ± {stats['score_std']:.3f}")
    print("=" * 60)
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(version=1)
    
    print("\n✅ Model saved successfully!")
    print(f"   Model: models/isolation_forest_v1.joblib")
    print(f"   Scaler: models/scaler_v1.joblib")
    print("\nYou can now run: sudo ./venv/bin/python3 main.py")
    print()


if __name__ == "__main__":
    main()
