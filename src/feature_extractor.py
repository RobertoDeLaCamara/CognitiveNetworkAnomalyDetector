"""Feature extraction module for ML-based anomaly detection.

This module extracts 18 features from network traffic packets to enable
machine learning-based anomaly detection. Features are categorized into:
- Statistical features (6)
- Temporal features (4)
- Protocol features (3)
- Port features (2)
- Payload features (3)
"""

import time
import math
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from scapy.all import IP, TCP, UDP, ICMP, Raw

from .ml_config import (
    FEATURE_WINDOW_SIZE,
    MAX_PACKET_HISTORY,
    FEATURE_NAMES,
    N_FEATURES,
    COMMON_PORTS
)


@dataclass
class PacketHistory:
    """Stores packet history for a single IP address."""
    
    # Raw packet data
    packet_sizes: deque = field(default_factory=lambda: deque(maxlen=MAX_PACKET_HISTORY))
    payload_sizes: deque = field(default_factory=lambda: deque(maxlen=MAX_PACKET_HISTORY))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=MAX_PACKET_HISTORY))
    protocols: deque = field(default_factory=lambda: deque(maxlen=MAX_PACKET_HISTORY))
    ports: deque = field(default_factory=lambda: deque(maxlen=MAX_PACKET_HISTORY))
    payload_entropies: deque = field(default_factory=lambda: deque(maxlen=MAX_PACKET_HISTORY))
    
    # Aggregated statistics
    first_seen: float = 0.0
    total_packets: int = 0
    total_bytes: int = 0
    tcp_count: int = 0
    udp_count: int = 0
    icmp_count: int = 0
    
    def add_packet(self, packet, timestamp: float):
        """Add a packet to the history."""
        if self.first_seen == 0.0:
            self.first_seen = timestamp
        
        # Extract packet information
        packet_size = len(packet)
        self.packet_sizes.append(packet_size)
        self.timestamps.append(timestamp)
        self.total_packets += 1
        self.total_bytes += packet_size
        
        # Protocol tracking
        protocol = None
        port = None
        
        if TCP in packet:
            protocol = 'TCP'
            port = packet[TCP].dport
            self.tcp_count += 1
        elif UDP in packet:
            protocol = 'UDP'
            port = packet[UDP].dport
            self.udp_count += 1
        elif ICMP in packet:
            protocol = 'ICMP'
            self.icmp_count += 1
        
        self.protocols.append(protocol)
        self.ports.append(port)
        
        # Payload analysis
        if Raw in packet:
            payload = packet[Raw].load
            payload_size = len(payload)
            entropy = self._calculate_entropy(payload)
        else:
            payload_size = 0
            entropy = 0.0
        
        self.payload_sizes.append(payload_size)
        self.payload_entropies.append(entropy)
    
    @staticmethod
    def _calculate_entropy(data: bytes) -> float:
        """Calculate Shannon entropy of byte data."""
        if len(data) == 0:
            return 0.0
        
        # Count byte frequencies
        byte_counts = defaultdict(int)
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        for count in byte_counts.values():
            probability = count / data_len
            entropy -= probability * math.log2(probability)
        
        return entropy


class FeatureExtractor:
    """Extracts ML features from network traffic packets."""
    
    def __init__(self, window_size: int = FEATURE_WINDOW_SIZE):
        """Initialize the feature extractor.
        
        Args:
            window_size: Time window in seconds for temporal features
        """
        self.window_size = window_size
        self.ip_data: Dict[str, PacketHistory] = defaultdict(PacketHistory)
        self.start_time = time.time()
    
    def process_packet(self, packet) -> Optional[str]:
        """Process a packet and update internal state.
        
        Args:
            packet: Scapy packet object
        
        Returns:
            IP address of the source, or None if packet has no IP layer
        """
        if IP not in packet:
            return None
        
        ip_src = packet[IP].src
        timestamp = time.time()
        
        # Add packet to history
        self.ip_data[ip_src].add_packet(packet, timestamp)
        
        return ip_src
    
    def extract_features(self, ip: str) -> Optional[np.ndarray]:
        """Extract feature vector for a given IP address.
        
        Args:
            ip: IP address to extract features for
        
        Returns:
            Numpy array of shape (N_FEATURES,) or None if insufficient data
        """
        if ip not in self.ip_data:
            return None
        
        history = self.ip_data[ip]
        
        # Need at least a few packets for meaningful features
        if history.total_packets < 3:
            return None
        
        try:
            features = []
            
            # ===== Statistical Features (6) =====
            features.extend(self._extract_statistical_features(history))
            
            # ===== Temporal Features (4) =====
            features.extend(self._extract_temporal_features(history))
            
            # ===== Protocol Features (3) =====
            features.extend(self._extract_protocol_features(history))
            
            # ===== Port Features (2) =====
            features.extend(self._extract_port_features(history))
            
            # ===== Payload Features (3) =====
            features.extend(self._extract_payload_features(history))
            
            # Verify feature count
            if len(features) != N_FEATURES:
                raise ValueError(f"Expected {N_FEATURES} features, got {len(features)}")
            
            return np.array(features, dtype=np.float64)
        
        except Exception as e:
            # Return None if feature extraction fails
            return None
    
    def _extract_statistical_features(self, history: PacketHistory) -> List[float]:
        """Extract statistical features (6 features)."""
        current_time = time.time()
        session_duration = current_time - history.first_seen
        
        # Avoid division by zero
        session_duration = max(session_duration, 0.001)
        
        # Calculate rates
        packets_per_second = history.total_packets / session_duration
        bytes_per_second = history.total_bytes / session_duration
        
        # Packet size statistics
        packet_sizes = list(history.packet_sizes)
        avg_packet_size = np.mean(packet_sizes) if packet_sizes else 0.0
        packet_size_variance = np.var(packet_sizes) if len(packet_sizes) > 1 else 0.0
        
        return [
            packets_per_second,
            bytes_per_second,
            avg_packet_size,
            packet_size_variance,
            float(history.total_packets),
            float(history.total_bytes),
        ]
    
    def _extract_temporal_features(self, history: PacketHistory) -> List[float]:
        """Extract temporal features (4 features)."""
        current_time = time.time()
        session_duration = current_time - history.first_seen
        
        # Calculate inter-arrival times
        timestamps = list(history.timestamps)
        inter_arrivals = []
        for i in range(1, len(timestamps)):
            inter_arrivals.append(timestamps[i] - timestamps[i-1])
        
        if inter_arrivals:
            inter_arrival_mean = np.mean(inter_arrivals)
            inter_arrival_std = np.std(inter_arrivals) if len(inter_arrivals) > 1 else 0.0
        else:
            inter_arrival_mean = 0.0
            inter_arrival_std = 0.0
        
        # Burst rate: packets in last 5 seconds
        recent_threshold = current_time - 5.0
        recent_packets = sum(1 for ts in timestamps if ts > recent_threshold)
        burst_rate = recent_packets / 5.0
        
        return [
            inter_arrival_mean,
            inter_arrival_std,
            burst_rate,
            session_duration,
        ]
    
    def _extract_protocol_features(self, history: PacketHistory) -> List[float]:
        """Extract protocol features (3 features)."""
        total = history.total_packets
        
        if total == 0:
            return [0.0, 0.0, 0.0]
        
        tcp_ratio = history.tcp_count / total
        udp_ratio = history.udp_count / total
        icmp_ratio = history.icmp_count / total
        
        return [tcp_ratio, udp_ratio, icmp_ratio]
    
    def _extract_port_features(self, history: PacketHistory) -> List[float]:
        """Extract port features (2 features)."""
        ports = [p for p in history.ports if p is not None]
        
        if not ports:
            return [0.0, 0.0]
        
        # Unique ports contacted
        unique_ports = len(set(ports))
        
        # Uncommon port ratio
        uncommon_count = sum(1 for p in ports if p not in COMMON_PORTS)
        uncommon_port_ratio = uncommon_count / len(ports)
        
        return [float(unique_ports), uncommon_port_ratio]
    
    def _extract_payload_features(self, history: PacketHistory) -> List[float]:
        """Extract payload features (3 features)."""
        payload_sizes = [s for s in history.payload_sizes if s > 0]
        payload_entropies = [e for e in history.payload_entropies if e > 0]
        
        # Average payload entropy
        avg_entropy = np.mean(payload_entropies) if payload_entropies else 0.0
        
        # Payload size statistics
        avg_payload_size = np.mean(payload_sizes) if payload_sizes else 0.0
        payload_size_variance = np.var(payload_sizes) if len(payload_sizes) > 1 else 0.0
        
        return [avg_entropy, avg_payload_size, payload_size_variance]
    
    def get_all_ips(self) -> List[str]:
        """Get list of all tracked IP addresses."""
        return list(self.ip_data.keys())
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return FEATURE_NAMES.copy()
    
    def reset(self):
        """Reset all stored data."""
        self.ip_data.clear()
        self.start_time = time.time()
