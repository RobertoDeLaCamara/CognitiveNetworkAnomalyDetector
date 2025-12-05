import pytest
from unittest.mock import MagicMock, patch
from scapy.all import IP, ICMP, TCP, UDP, Raw
from src.anomaly_detector import packet_analyzer

# Mock the logger to prevent it from writing to a file during tests
@pytest.fixture(autouse=True)
def mock_logger():
    """
    Fixture to mock the logger so that it doesn't write to a file during tests.
    
    This fixture is marked as autouse=True, meaning it will be automatically used by all tests.
    """
    with patch('src.anomaly_detector.logger') as mock_log:
        yield mock_log

# Test for ICMP flood detection
def test_icmp_flood_detection(mocker):
    """
    Tests that an ICMP flood is correctly detected when the packet count exceeds the threshold.
    """
    # Create a fresh analyzer for this test
    analyzer = packet_analyzer
    analyzer.packet_count_per_ip['10.0.0.1'] = 100
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(10.0, 2.0))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    # Create a mock ICMP packet
    packet = IP(src='10.0.0.1')/ICMP()
    analyzer.analyze_packet(packet)

    # Check if the alert was logged
    mock_log_alert.assert_called_with(
        "ALERT: Possible ICMP attack (ping flood) from 10.0.0.1",
        "IP 10.0.0.1 has sent more than 50 ICMP packets."
    )

# Test for traffic spike detection
def test_traffic_spike_detection(mocker):
    """
    Tests that a traffic spike is detected when the current rate is much higher than the average.
    """
    analyzer = packet_analyzer
    analyzer.packet_count_per_ip['10.0.0.2'] = 200
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(50.0, 5.0))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    packet = IP(src='10.0.0.2')/TCP()
    analyzer.analyze_packet(packet)

    mock_log_alert.assert_called_with(
        "ALERT: Traffic spike from 10.0.0.2",
        "IP 10.0.0.2 has a traffic rate of 50.00 packets/sec, which is significantly higher than the average of 5.00 packets/sec."
    )

# Test for unusual port traffic
def test_unusual_port_traffic(mocker):
    """
    Tests that traffic to an uncommon port triggers an alert.
    """
    analyzer = packet_analyzer
    analyzer.packet_count_per_ip['10.0.0.3'] = 10
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(2.0, 1.0))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    packet = IP(src='10.0.0.3')/UDP(dport=12345)
    analyzer.analyze_packet(packet)

    mock_log_alert.assert_called_with(
        "ALERT: Traffic on uncommon port 12345 from 10.0.0.3",
        "Traffic detected from IP 10.0.0.3 to port 12345, which is unusual."
    )

# Test for large payload detection
def test_large_payload_detection(mocker):
    """
    Tests that a packet with an unusually large payload triggers an alert.
    """
    analyzer = packet_analyzer
    analyzer.packet_count_per_ip['10.0.0.4'] = 5
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(1.0, 0.5))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    # Create a payload larger than the threshold
    large_payload = b'A' * 200
    packet = IP(src='10.0.0.4')/TCP(dport=80)/Raw(load=large_payload)
    analyzer.analyze_packet(packet)

    mock_log_alert.assert_called_with(
        "ALERT: Unusually large payload from 10.0.0.4",
        "A payload of 200 bytes was detected from 10.0.0.4 to port 80."
    )

# Test for malicious payload detection in a packet
def test_malicious_payload_in_packet(mocker):
    """
    Tests that a malicious payload within a packet triggers the correct alert.
    """
    analyzer = packet_analyzer
    analyzer.packet_count_per_ip['10.0.0.5'] = 3
    
    mocker.patch.object(analyzer, 'calculate_packet_rate', return_value=(1.0, 1.0))
    mock_log_alert = mocker.patch.object(analyzer, 'log_alert')

    malicious_payload = b"UNION SELECT user, password FROM users"
    packet = IP(src='10.0.0.5')/TCP(dport=443)/Raw(load=malicious_payload)
    analyzer.analyze_packet(packet)

    # Now with longest-first pattern matching, "UNION SELECT" is correctly matched
    mock_log_alert.assert_called_with(
        "ALERT: Malicious payload detected from 10.0.0.5",
        "The pattern 'UNION SELECT' was detected in traffic from 10.0.0.5 to port 443."
    )
