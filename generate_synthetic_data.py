#!/usr/bin/env python3
"""Generate synthetic training data for testing the anomaly detector."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import argparse

# Add src to path securely
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir / 'src'))

from src.ml_config import N_FEATURES, FEATURE_NAMES

def generate_synthetic_data(n_samples=150):
    """Generate synthetic network traffic features.
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with synthetic features
        
    Raises:
        ValueError: If n_samples is invalid
    """
    if not isinstance(n_samples, int) or n_samples <= 0 or n_samples > 100000:
        raise ValueError("n_samples must be between 1 and 100000")
    
    np.random.seed(42)
    
    # Generate normal traffic patterns
    data = {}
    
    # Statistical features (packets and bytes)
    data['packets_per_second'] = np.random.exponential(scale=10, size=n_samples)
    data['bytes_per_second'] = data['packets_per_second'] * np.random.normal(500, 100, n_samples)
    data['avg_packet_size'] = np.random.normal(500, 100, n_samples)
    data['packet_size_variance'] = np.random.exponential(scale=1000, size=n_samples)
    data['total_packets'] = np.random.poisson(lam=100, size=n_samples)
    data['total_bytes'] = data['total_packets'] * data['avg_packet_size']
    
    # Temporal features
    data['inter_arrival_mean'] = np.random.exponential(scale=0.1, size=n_samples)
    data['inter_arrival_std'] = data['inter_arrival_mean'] * np.random.uniform(0.1, 0.5, n_samples)
    data['burst_rate'] = np.random.uniform(0, 5, n_samples)
    data['session_duration'] = np.random.exponential(scale=60, size=n_samples)
    
    # Protocol features (ratios sum to 1)
    tcp_ratio = np.random.beta(5, 2, n_samples)  # Most traffic is TCP
    udp_ratio = np.random.beta(2, 5, n_samples) * (1 - tcp_ratio)
    icmp_ratio = 1 - tcp_ratio - udp_ratio
    
    data['tcp_ratio'] = tcp_ratio
    data['udp_ratio'] = udp_ratio
    data['icmp_ratio'] = icmp_ratio
    
    # Port features
    data['unique_ports'] = np.random.poisson(lam=5, size=n_samples)
    data['uncommon_port_ratio'] = np.random.uniform(0, 0.3, n_samples)
    
    # Payload features
    data['payload_entropy'] = np.random.uniform(0.3, 0.9, n_samples)  # Normal entropy range
    data['avg_payload_size'] = np.random.normal(300, 80, n_samples)
    data['payload_size_variance'] = np.random.exponential(scale=500, size=n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add IP addresses
    df.insert(0, 'ip', [f"192.168.1.{i % 254 + 1}" for i in range(n_samples)])
    
    # Ensure all values are positive
    for col in df.columns:
        if col != 'ip':
            df[col] = df[col].clip(lower=0)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument('--samples', type=int, default=150, help='Number of samples (1-100000)')
    parser.add_argument('--output', type=str, default='data/training/synthetic_baseline.csv', help='Output file')
    args = parser.parse_args()
    
    try:
        print("Generating synthetic training data...")
        df = generate_synthetic_data(n_samples=args.samples)
        
        output_file = Path(args.output).resolve()
        if '..' in str(args.output):
            print("Error: Path traversal detected")
            sys.exit(1)
        
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, mode=0o750, exist_ok=True)
        df.to_csv(output_file, index=False)
        output_file.chmod(0o640)
        
        print(f"✅ Generated {len(df)} samples")
        print(f"   Saved to: {output_file}")
        print(f"   Features: {len(df.columns) - 1}")
        print(f"\nYou can now train with:")
        print(f"   python train_model.py --from-file {output_file} --version 1")
    except (ValueError, Exception) as e:
        print(f"Error: {e}")
        sys.exit(1)
