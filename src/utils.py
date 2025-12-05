"""Utility functions for the anomaly detection system."""

import math
from typing import Dict


def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte data.
    
    Shannon entropy measures the randomness/unpredictability of data.
    For byte data, the maximum entropy is 8 bits (perfectly random).
    
    Args:
        data: Byte sequence to calculate entropy for
    
    Returns:
        Shannon entropy value (0-8 for byte data)
        - 0.0 = no randomness (all same byte)
        - 8.0 = maximum randomness (uniform distribution)
    
    Examples:
        >>> calculate_entropy(b'AAAA')
        0.0
        >>> calculate_entropy(bytes(range(256)))  # doctest: +SKIP
        8.0
    """
    if len(data) == 0:
        return 0.0
    
    # Count byte frequencies
    byte_counts: Dict[int, int] = {}
    for byte in data:
        byte_counts[byte] = byte_counts.get(byte, 0) + 1
    
    # Calculate Shannon entropy
    entropy = 0.0
    data_len = len(data)
    
    for count in byte_counts.values():
        if count > 0:
            probability = count / data_len
            entropy -= probability * math.log2(probability)
    
    return entropy
