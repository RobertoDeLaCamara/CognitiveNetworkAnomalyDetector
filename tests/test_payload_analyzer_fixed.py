import pytest
from src.payload_analyzer import detect_malicious_payload

# Test cases for malicious payloads that actually exist in our config
@pytest.mark.parametrize("payload, expected_pattern", [
    # Working patterns from our security config
    (b"eval(base64_decode('test'))", "eval(base64_decode("),
    (b"../../../etc/passwd", "../../../etc/passwd"),
    (b"admin'--", "admin'--"),
    (b"xp_cmdshell", "xp_cmdshell"),
    (b"'; DROP TABLE users", "'; DROP TABLE"),
    (b"' OR '1'='1'", "' OR '1'='1'"),
    (b"UNION SELECT * FROM users", "UNION SELECT"),
])
def test_detect_malicious_payload_working_patterns(payload, expected_pattern):
    """
    Tests that the detect_malicious_payload function correctly identifies 
    malicious patterns that actually exist in our configuration.
    """
    is_malicious, pattern = detect_malicious_payload(payload)
    assert is_malicious is True
    assert pattern == expected_pattern

# Test cases for benign payloads
@pytest.mark.parametrize("payload", [
    b"Hello, this is a normal message.",
    b"Here is a file transfer with no malicious content.",
    b"Just a regular HTTP request.",
    b"SELECTion of products is great",  # Benign case that contains a keyword
    b"This is a normal script tag",
    b"Regular file path /home/user/file.txt",
])
def test_detect_malicious_payload_benign(payload):
    """
    Tests that the detect_malicious_payload function does not flag benign payloads.
    """
    is_malicious, pattern = detect_malicious_payload(payload)
    assert is_malicious is False
    assert pattern is None

# Test with empty payload
def test_detect_malicious_payload_empty():
    """
    Tests that an empty payload is correctly identified as not malicious.
    """
    is_malicious, pattern = detect_malicious_payload(b"")
    assert is_malicious is False
    assert pattern is None

# Test with None payload
def test_detect_malicious_payload_none():
    """
    Tests that None payload is handled gracefully.
    """
    is_malicious, pattern = detect_malicious_payload(None)
    assert is_malicious is False
    assert pattern is None

# Test with oversized payload
def test_detect_malicious_payload_oversized():
    """
    Tests that oversized payloads are handled according to security limits.
    """
    large_payload = b"A" * 2000  # Larger than MAX_PAYLOAD_SCAN_SIZE
    is_malicious, pattern = detect_malicious_payload(large_payload)
    # Should return False due to size limit
    assert is_malicious is False
    assert pattern is None

# Test case sensitivity
def test_detect_malicious_payload_case_insensitive():
    """
    Tests that pattern matching is case insensitive.
    """
    # Test uppercase version of a known pattern
    is_malicious, pattern = detect_malicious_payload(b"UNION select * from users")
    assert is_malicious is True
    assert pattern == "UNION SELECT"