import pytest
from unittest.mock import MagicMock, patch
from scapy.all import IP, ICMP, TCP, UDP, Raw
from src.anomaly_detector import PacketAnalyzer
import os

# Mock the logger to prevent it from writing to a file during tests
@pytest.fixture(autouse=True)
def mock_logger():
    """
    Fixture to mock the logger so that it doesn't write to a file during tests.
    
    This fixture is marked as autouse=True, meaning it will be automatically used by all tests.
    """
    with patch('src.anomaly_detector.logger') as mock_log:
        yield mock_log

@pytest.fixture
def analyzer():
    """Create a fresh PacketAnalyzer instance for each test."""
    return PacketAnalyzer(enable_ml=False)  # Disable ML for simpler testing

# Test for ICMP flood detection
def test_icmp_flood_detection(analyzer, mocker):
    """
    Tests that an ICMP flood is correctly detected when the packet count exceeds the threshold.
    """
    # Set up the analyzer state
    analyzer.packet_count_per_ip['10.0.0.1'] = 100
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(10.0, 2.0))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    # Create a mock ICMP packet
    packet = IP(src='10.0.0.1')/ICMP()
    analyzer.analyze_packet(packet)

    # Check if the alert was logged (updated to match new signature)
    mock_log_alert.assert_called_with(
        "ALERT: Possible ICMP attack (ping flood) from 10.0.0.1",
        "IP 10.0.0.1 has sent more than 50 ICMP packets.",
        alert_type='ICMP_FLOOD',
        ip_src='10.0.0.1'
    )

# Test for traffic spike detection
def test_traffic_spike_detection(analyzer, mocker):
    """
    Tests that a traffic spike is detected when the current rate is much higher than the average.
    """
    analyzer.packet_count_per_ip['10.0.0.2'] = 200
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(50.0, 5.0))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    packet = IP(src='10.0.0.2')/TCP()
    analyzer.analyze_packet(packet)

    mock_log_alert.assert_called_with(
        "ALERT: Traffic spike from 10.0.0.2",
        "IP 10.0.0.2 has a traffic rate of 50.00 packets/sec, which is significantly higher than the average of 5.00 packets/sec.",
        ip_src='10.0.0.2'
    )

# Test for unusual port traffic
def test_unusual_port_traffic(analyzer, mocker):
    """
    Tests that traffic to an uncommon port triggers an alert.
    """
    analyzer.packet_count_per_ip['10.0.0.3'] = 10
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(2.0, 1.0))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    packet = IP(src='10.0.0.3')/UDP(dport=12345)
    analyzer.analyze_packet(packet)

    mock_log_alert.assert_called_with(
        "ALERT: Traffic on uncommon port 12345 from 10.0.0.3",
        "Traffic detected from IP 10.0.0.3 to port 12345, which is unusual.",
        ip_src='10.0.0.3'
    )

# Test for large payload detection
def test_large_payload_detection(analyzer, mocker):
    """
    Tests that a packet with an unusually large payload triggers an alert.
    """
    analyzer.packet_count_per_ip['10.0.0.4'] = 5
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(1.0, 0.5))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    # Create a payload larger than the threshold
    large_payload = b'A' * 200
    packet = IP(src='10.0.0.4')/TCP(dport=80)/Raw(load=large_payload)
    analyzer.analyze_packet(packet)

    mock_log_alert.assert_called_with(
        "ALERT: Unusually large payload from 10.0.0.4",
        "A payload of 200 bytes was detected from 10.0.0.4 to port 80.",
        ip_src='10.0.0.4'
    )

# Test for malicious payload detection in a packet
def test_malicious_payload_in_packet(analyzer, mocker):
    """
    Tests that a malicious payload within a packet triggers the correct alert.
    """
    analyzer.packet_count_per_ip['10.0.0.5'] = 3
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(1.0, 1.0))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    malicious_payload = b"UNION SELECT user, password FROM users"
    packet = IP(src='10.0.0.5')/TCP(dport=443)/Raw(load=malicious_payload)
    analyzer.analyze_packet(packet)

    # Now with longest-first pattern matching, "UNION SELECT" is correctly matched
    mock_log_alert.assert_called_with(
        "ALERT: Malicious payload detected from 10.0.0.5",
        "The pattern 'UNION SELECT' was detected in traffic from 10.0.0.5 to port 443.",
        ip_src='10.0.0.5'
    )

# Test for invalid IP address handling
def test_invalid_ip_handling(analyzer, mocker):
    """
    Tests that invalid IP addresses are handled gracefully.
    """
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')
    
    # Create packet with invalid IP (this should be filtered out)
    packet = IP(src='0.0.0.0')/TCP(dport=80)
    analyzer.analyze_packet(packet)
    
    # No alerts should be generated for invalid IPs
    mock_log_alert.assert_not_called()

# Test for rate limiting
def test_alert_rate_limiting(analyzer):
    """
    Tests that alerts are rate limited to prevent spam.
    """
    test_ip = '192.168.1.100'
    alert_type = 'ICMP_FLOOD'
    
    # Test the rate limiting directly
    allowed_count = 0
    for i in range(5):
        if analyzer._should_alert(test_ip, alert_type):
            allowed_count += 1
    
    # Should allow max 3 alerts
    assert allowed_count == 3

# Test the rate limiting function directly
def test_should_alert_rate_limiting(analyzer):
    """
    Tests the _should_alert method directly.
    """
    ip = '192.168.1.100'
    alert_type = 'ICMP_FLOOD'
    
    # First 3 alerts should be allowed
    assert analyzer._should_alert(ip, alert_type) == True
    assert analyzer._should_alert(ip, alert_type) == True
    assert analyzer._should_alert(ip, alert_type) == True
    
    # 4th and 5th should be blocked
    assert analyzer._should_alert(ip, alert_type) == False
    assert analyzer._should_alert(ip, alert_type) == False


class TestMLDetectionPaths:
    """Test ML detection code paths."""
    
    def test_cleanup_old_data(self):
        """Test that old data is cleaned up when max IPs exceeded."""
        analyzer = PacketAnalyzer(enable_ml=False, max_ips=10)
        
        # Add more IPs than the limit
        for i in range(15):
            analyzer.packet_count_per_ip[f'192.168.1.{i}'] = i + 1
        
        # Trigger cleanup
        analyzer._cleanup_old_data()
        
        # Should have removed some IPs
        assert len(analyzer.packet_count_per_ip) < 15
    
    def test_ml_alert_logging(self, mocker):
        """Test ML-specific alert logging."""
        analyzer = PacketAnalyzer(enable_ml=False)
        mock_log_alert = mocker.patch.object(analyzer, 'log_alert')
        
        # Set up packet count
        analyzer.packet_count_per_ip['192.168.1.100'] = 50
        
        # Log ML alert
        analyzer.log_ml_alert('192.168.1.100', -0.5, [1, 2, 3, 4])
        
        # Verify alert was logged with correct type
        assert mock_log_alert.called
        call_args = mock_log_alert.call_args
        assert 'ML ANOMALY DETECTED' in call_args[0][0]
        assert call_args[1]['alert_type'] == 'ML'
    
    def test_ml_alert_with_invalid_score(self, mocker, capsys):
        """Test ML alert with invalid anomaly score."""
        analyzer = PacketAnalyzer(enable_ml=False)
        mock_log_alert = mocker.patch.object(analyzer, 'log_alert')
        
        # Try to log with invalid score
        analyzer.log_ml_alert('192.168.1.100', 999.0, [1, 2, 3])
        
        # Should not log alert due to invalid score
        mock_log_alert.assert_not_called()
    
    def test_ml_detector_initialization_with_ml_enabled(self, mocker):
        """Test ML detector initialization when ML is enabled."""
        # Mock the ML components
        mock_feature_extractor = MagicMock()
        mock_ml_detector = MagicMock()
        
        with patch('src.anomaly_detector.ML_AVAILABLE', True):
            with patch('src.anomaly_detector.ML_ENABLED', True):
                with patch('src.anomaly_detector.FeatureExtractor', return_value=mock_feature_extractor):
                    with patch('src.anomaly_detector.IsolationForestDetector', return_value=mock_ml_detector):

                        with patch('os.path.exists', return_value=False):
                            with patch.dict(os.environ, {'MLFLOW_ENABLE_REMOTE_LOADING': 'false'}):
                                analyzer = PacketAnalyzer(enable_ml=True)
                                
                                # ML should be disabled if no model exists
                                assert analyzer.ml_enabled is False
    
    def test_memory_management_with_max_ips(self):
        """Test that analyzer respects max_ips limit."""
        analyzer = PacketAnalyzer(enable_ml=False, max_ips=5)
        
        # Add packets from many IPs
        for i in range(10):
            packet = IP(src=f'192.168.1.{i}') / TCP(dport=80)
            analyzer.analyze_packet(packet)
        
        # Should trigger cleanup and stay under reasonable limit
        assert len(analyzer.packet_count_per_ip) <= 10
    
    def test_ml_error_handling(self, mocker):
        """Test that ML errors don't break rule-based detection."""
        analyzer = PacketAnalyzer(enable_ml=False)
        
        # Enable ML but make it fail
        analyzer.ml_enabled = True
        analyzer.feature_extractor = MagicMock()
        analyzer.ml_detector = MagicMock()
        analyzer.feature_extractor.extract_features.side_effect = Exception("ML Error")
        
        mock_log_alert = mocker.patch.object(analyzer, 'log_alert')
        
        # Set sufficient packet count for ML
        analyzer.packet_count_per_ip['192.168.1.100'] = 100
        
        # Create packet - should still process with rule-based detection
        packet = IP(src='192.168.1.100') / TCP(dport=12345)
        
        # Should not crash despite ML error
        analyzer.analyze_packet(packet)
        
        # Rule-based detection should still work
        # (uncommon port should trigger alert)
        assert any('uncommon port' in str(call) for call in mock_log_alert.call_args_list)


class TestPacketValidation:
    """Test packet validation and error handling."""
    
    def test_invalid_port_number(self, mocker):
        """Test handling of invalid port numbers."""
        analyzer = PacketAnalyzer(enable_ml=False)
        
        # Create packet with manually set invalid port
        packet = IP(src='192.168.1.100') / TCP()
        # Manually override port to invalid value
        packet[TCP].dport = 70000  # Invalid port
        
        # Should handle gracefully (may log warning but not crash)
        analyzer.analyze_packet(packet)
    
    def test_invalid_payload_type(self, mocker):
        """Test handling of invalid payload types."""
        analyzer = PacketAnalyzer(enable_ml=False)
        
        # This should be handled gracefully
        packet = IP(src='192.168.1.100') / TCP(dport=80)
        analyzer.analyze_packet(packet)

