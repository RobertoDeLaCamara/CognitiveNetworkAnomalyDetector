"""Unit tests for the feature extraction module."""

import pytest
import numpy as np
from unittest.mock import MagicMock
from scapy.all import IP, TCP, UDP, ICMP, Raw

import sys
sys.path.insert(0, 'src')

from src.feature_extractor import FeatureExtractor, PacketHistory
from src.ml_config import N_FEATURES


class TestPacketHistory:
    """Tests for PacketHistory class."""
    
    def test_initialization(self):
        """Test PacketHistory initialization."""
        history = PacketHistory()
        assert history.first_seen == 0.0
        assert history.total_packets == 0
        assert history.total_bytes == 0
        assert len(history.packet_sizes) == 0
    
    def test_entropy_calculation(self):
        """Test Shannon entropy calculation."""
        from src.utils import calculate_entropy
        
        # Uniform distribution (max entropy)
        uniform_data = bytes(range(256))
        entropy = calculate_entropy(uniform_data)
        assert entropy == pytest.approx(8.0, abs=0.1)  # ~8 bits for uniform
        
        # All same byte (min entropy)
        uniform_byte = b'A' * 100
        entropy = calculate_entropy(uniform_byte)
        assert entropy == 0.0
        
        # Empty data
        entropy = calculate_entropy(b'')
        assert entropy == 0.0


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a feature extractor instance."""
        return FeatureExtractor(window_size=60)
    
    def test_initialization(self, extractor):
        """Test FeatureExtractor initialization."""
        assert extractor.window_size == 60
        assert len(extractor.ip_data) == 0
    
    def test_process_packet_with_ip(self, extractor):
        """Test processing a valid IP packet."""
        packet = IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=80)
        ip = extractor.process_packet(packet)
        
        assert ip == "192.168.1.100"
        assert "192.168.1.100" in extractor.ip_data
    
    def test_process_packet_without_ip(self, extractor):
        """Test processing a packet without IP layer."""
        packet = MagicMock()
        packet.__contains__ = MagicMock(return_value=False)
        
        ip = extractor.process_packet(packet)
        assert ip is None
    
    def test_extract_features_insufficient_data(self, extractor):
        """Test feature extraction with insufficient packets."""
        # Add only 1 packet
        packet = IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=80)
        extractor.process_packet(packet)
        
        features = extractor.extract_features("192.168.1.100")
        assert features is None  # Not enough packets
    
    def test_extract_features_valid(self, extractor):
        """Test feature extraction with sufficient packets."""
        # Add multiple packets
        for i in range(10):
            packet = IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=80) / Raw(load=b"test payload")
            extractor.process_packet(packet)
        
        features = extractor.extract_features("192.168.1.100")
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.shape == (N_FEATURES,)
        assert not np.any(np.isnan(features))  # No NaN values
        assert not np.any(np.isinf(features))  # No inf values
    
    def test_feature_count(self, extractor):
        """Test that correct number of features is extracted."""
        # Add multiple packets with variety
        packets = [
            IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=80) / Raw(load=b"http payload"),
            IP(src="192.168.1.100", dst="192.168.1.1") / UDP(dport=53) / Raw(load=b"dns query"),
            IP(src="192.168.1.100", dst="192.168.1.1") / ICMP(),
            IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=443),
            IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=8080) / Raw(load=b"test"),
        ]
        
        for packet in packets:
            extractor.process_packet(packet)
        
        features = extractor.extract_features("192.168.1.100")
        assert len(features) == N_FEATURES
    
    def test_protocol_features(self, extractor):
        """Test protocol ratio features."""
        # Add TCP, UDP, and ICMP packets
        extractor.process_packet(IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=80))
        extractor.process_packet(IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=443))
        extractor.process_packet(IP(src="192.168.1.100", dst="192.168.1.1") / UDP(dport=53))
        extractor.process_packet(IP(src="192.168.1.100", dst="192.168.1.1") / ICMP())
        
        features = extractor.extract_features("192.168.1.100")
        
        # Protocol features are at indices 10, 11, 12
        tcp_ratio = features[10]
        udp_ratio = features[11]
        icmp_ratio = features[12]
        
        assert tcp_ratio == pytest.approx(0.5)  # 2/4 packets
        assert udp_ratio == pytest.approx (0.25)  # 1/4 packets
        assert icmp_ratio == pytest.approx(0.25)  # 1/4 packets
        assert tcp_ratio + udp_ratio + icmp_ratio == pytest.approx(1.0)
    
    def test_get_all_ips(self, extractor):
        """Test getting all tracked IPs."""
        ips = ["192.168.1.100", "192.168.1.101", "192.168.1.102"]
        
        for ip in ips:
            packet = IP(src=ip, dst="192.168.1.1") / TCP(dport=80)
            extractor.process_packet(packet)
        
        tracked_ips = extractor.get_all_ips()
        assert set(tracked_ips) == set(ips)
    
    def test_reset(self, extractor):
        """Test resetting the extractor."""
        # Add some packets
        for i in range(5):
            packet = IP(src="192.168.1.100", dst="192.168.1.1") / TCP(dport=80)
            extractor.process_packet(packet)
        
        assert len(extractor.ip_data) > 0
        
        # Reset
        extractor.reset()
        
        assert len(extractor.ip_data) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
